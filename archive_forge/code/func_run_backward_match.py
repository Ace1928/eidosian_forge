import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def run_backward_match(self):
    """Run the backward match algorithm and returns the list of matches given an initial match, a forward
        scenario and a circuit qubits configuration.
        """
    match_store_list = []
    counter = 1
    pattern_blocked = [False] * self.pattern_dag.size
    first_match = MatchingScenarios(self.circuit_matched, self.circuit_blocked, self.pattern_matched, pattern_blocked, self.forward_matches, counter)
    self.matching_list = MatchingScenariosList()
    self.matching_list.append_scenario(first_match)
    gate_indices = _gate_indices(self.circuit_matched, self.circuit_blocked)
    number_of_gate_to_match = self.pattern_dag.size - (self.node_id_p - 1) - len(self.forward_matches)
    while self.matching_list.matching_scenarios_list:
        scenario = self.matching_list.pop_scenario()
        circuit_matched = scenario.circuit_matched
        circuit_blocked = scenario.circuit_blocked
        pattern_matched = scenario.pattern_matched
        pattern_blocked = scenario.pattern_blocked
        matches_scenario = scenario.matches
        counter_scenario = scenario.counter
        match_backward = [match for match in matches_scenario if match not in self.forward_matches]
        if counter_scenario > len(gate_indices) or len(match_backward) == number_of_gate_to_match:
            matches_scenario.sort(key=lambda x: x[0])
            match_store_list.append(Match(matches_scenario, self.qubits_conf))
            continue
        circuit_id = gate_indices[counter_scenario - 1]
        node_circuit = self.circuit_dag.get_node(circuit_id)
        if circuit_blocked[circuit_id]:
            matching_scenario = MatchingScenarios(circuit_matched, circuit_blocked, pattern_matched, pattern_blocked, matches_scenario, counter_scenario + 1)
            self.matching_list.append_scenario(matching_scenario)
            continue
        candidates_indices = self._find_backward_candidates(pattern_blocked, matches_scenario)
        wires1 = self.wires[circuit_id]
        control_wires1 = self.control_wires[circuit_id]
        target_wires1 = self.target_wires[circuit_id]
        global_match = False
        global_broken = []
        for pattern_id in candidates_indices:
            node_pattern = self.pattern_dag.get_node(pattern_id)
            wires2 = self.pattern_dag.get_node(pattern_id).wires
            control_wires2 = self.pattern_dag.get_node(pattern_id).control_wires
            target_wires2 = self.pattern_dag.get_node(pattern_id).target_wires
            if len(wires1) != len(wires2) or set(wires1) != set(wires2) or node_circuit.op.name != node_pattern.op.name:
                continue
            if _compare_operation_without_qubits(node_circuit, node_pattern):
                if _compare_qubits(node_circuit, wires1, control_wires1, target_wires1, wires2, control_wires2, target_wires2):
                    circuit_matched_match = circuit_matched.copy()
                    circuit_blocked_match = circuit_blocked.copy()
                    pattern_matched_match = pattern_matched.copy()
                    pattern_blocked_match = pattern_blocked.copy()
                    matches_scenario_match = matches_scenario.copy()
                    block_list = []
                    broken_matches_match = []
                    for potential_block in self.pattern_dag.successors(pattern_id):
                        if not pattern_matched_match[potential_block]:
                            pattern_blocked_match[potential_block] = True
                            block_list.append(potential_block)
                            for block_id in block_list:
                                for succ_id in self.pattern_dag.successors(block_id):
                                    pattern_blocked_match[succ_id] = True
                                    if pattern_matched_match[succ_id]:
                                        new_id = pattern_matched_match[succ_id][0]
                                        circuit_matched_match[new_id] = []
                                        pattern_matched_match[succ_id] = []
                                        broken_matches_match.append(succ_id)
                    if broken_matches_match:
                        global_broken.append(True)
                    else:
                        global_broken.append(False)
                    new_matches_scenario_match = [elem for elem in matches_scenario_match if elem[0] not in broken_matches_match]
                    condition = True
                    for back_match in match_backward:
                        if back_match not in new_matches_scenario_match:
                            condition = False
                            break
                    if [self.node_id_p, self.node_id_c] in new_matches_scenario_match and (condition or not match_backward):
                        pattern_matched_match[pattern_id] = [circuit_id]
                        circuit_matched_match[circuit_id] = [pattern_id]
                        new_matches_scenario_match.append([pattern_id, circuit_id])
                        new_matching_scenario = MatchingScenarios(circuit_matched_match, circuit_blocked_match, pattern_matched_match, pattern_blocked_match, new_matches_scenario_match, counter_scenario + 1)
                        self.matching_list.append_scenario(new_matching_scenario)
                        global_match = True
        if global_match:
            circuit_matched_block_s = circuit_matched.copy()
            circuit_blocked_block_s = circuit_blocked.copy()
            pattern_matched_block_s = pattern_matched.copy()
            pattern_blocked_block_s = pattern_blocked.copy()
            matches_scenario_block_s = matches_scenario.copy()
            circuit_blocked_block_s[circuit_id] = True
            broken_matches = []
            for succ in self.circuit_dag.get_node(circuit_id).successors:
                circuit_blocked_block_s[succ] = True
                if circuit_matched_block_s[succ]:
                    broken_matches.append(succ)
                    new_id = circuit_matched_block_s[succ][0]
                    pattern_matched_block_s[new_id] = []
                    circuit_matched_block_s[succ] = []
            new_matches_scenario_block_s = [elem for elem in matches_scenario_block_s if elem[1] not in broken_matches]
            condition_not_greedy = True
            for back_match in match_backward:
                if back_match not in new_matches_scenario_block_s:
                    condition_not_greedy = False
                    break
            if [self.node_id_p, self.node_id_c] in new_matches_scenario_block_s and (condition_not_greedy or not match_backward):
                new_matching_scenario = MatchingScenarios(circuit_matched_block_s, circuit_blocked_block_s, pattern_matched_block_s, pattern_blocked_block_s, new_matches_scenario_block_s, counter_scenario + 1)
                self.matching_list.append_scenario(new_matching_scenario)
            if broken_matches and all(global_broken):
                circuit_matched_block_p = circuit_matched.copy()
                circuit_blocked_block_p = circuit_blocked.copy()
                pattern_matched_block_p = pattern_matched.copy()
                pattern_blocked_block_p = pattern_blocked.copy()
                matches_scenario_block_p = matches_scenario.copy()
                circuit_blocked_block_p[circuit_id] = True
                for pred in self.circuit_dag.get_node(circuit_id).predecessors:
                    circuit_blocked_block_p[pred] = True
                matching_scenario = MatchingScenarios(circuit_matched_block_p, circuit_blocked_block_p, pattern_matched_block_p, pattern_blocked_block_p, matches_scenario_block_p, counter_scenario + 1)
                self.matching_list.append_scenario(matching_scenario)
        if not global_match:
            circuit_blocked[circuit_id] = True
            following_matches = []
            successors = self.circuit_dag.get_node(circuit_id).successors
            for succ in successors:
                if circuit_matched[succ]:
                    following_matches.append(succ)
            predecessors = self.circuit_dag.get_node(circuit_id).predecessors
            if not predecessors or not following_matches:
                matching_scenario = MatchingScenarios(circuit_matched, circuit_blocked, pattern_matched, pattern_blocked, matches_scenario, counter_scenario + 1)
                self.matching_list.append_scenario(matching_scenario)
            else:
                circuit_matched_nomatch = circuit_matched.copy()
                circuit_blocked_nomatch = circuit_blocked.copy()
                pattern_matched_nomatch = pattern_matched.copy()
                pattern_blocked_nomatch = pattern_blocked.copy()
                matches_scenario_nomatch = matches_scenario.copy()
                for pred in predecessors:
                    circuit_blocked[pred] = True
                matching_scenario = MatchingScenarios(circuit_matched, circuit_blocked, pattern_matched, pattern_blocked, matches_scenario, counter_scenario + 1)
                self.matching_list.append_scenario(matching_scenario)
                broken_matches = []
                successors = self.circuit_dag.get_node(circuit_id).successors
                for succ in successors:
                    circuit_blocked_nomatch[succ] = True
                    if circuit_matched_nomatch[succ]:
                        broken_matches.append(succ)
                        circuit_matched_nomatch[succ] = []
                new_matches_scenario_nomatch = [elem for elem in matches_scenario_nomatch if elem[1] not in broken_matches]
                condition_block = True
                for back_match in match_backward:
                    if back_match not in new_matches_scenario_nomatch:
                        condition_block = False
                        break
                if [self.node_id_p, self.node_id_c] in matches_scenario_nomatch and (condition_block or not match_backward):
                    new_matching_scenario = MatchingScenarios(circuit_matched_nomatch, circuit_blocked_nomatch, pattern_matched_nomatch, pattern_blocked_nomatch, new_matches_scenario_nomatch, counter_scenario + 1)
                    self.matching_list.append_scenario(new_matching_scenario)
    length = max((len(m.match) for m in match_store_list))
    for scenario in match_store_list:
        if len(scenario.match) == length and (not any((scenario.match == x.match for x in self.match_final))):
            self.match_final.append(scenario)