from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine
def reduce_using_automaton(self, word):
    """
        Reduce a word using an automaton.

        Summary:
        All the symbols of the word are stored in an array and are given as the input to the automaton.
        If the automaton reaches a dead state that subword is replaced and the automaton is run from the beginning.
        The complete word has to be replaced when the word is read and the automaton reaches a dead state.
        So, this process is repeated until the word is read completely and the automaton reaches the accept state.

        Arguments:
            word (instance of FreeGroupElement) -- Word that needs to be reduced.

        """
    if self._new_rules:
        self._add_to_automaton(self._new_rules)
        self._new_rules = {}
    flag = 1
    while flag:
        flag = 0
        current_state = self.reduction_automaton.states['start']
        for i, s in enumerate(word.letter_form_elm):
            next_state_name = current_state.transitions[s]
            next_state = self.reduction_automaton.states[next_state_name]
            if next_state.state_type == 'd':
                subst = next_state.rh_rule
                word = word.substituted_word(i - len(next_state_name) + 1, i + 1, subst)
                flag = 1
                break
            current_state = next_state
    return word