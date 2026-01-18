from pyomo.common.dependencies import networkx as nx
from pyomo.core import Constraint, Objective, Var, ComponentMap, SortComponents
from pyomo.core.expr import identify_variables
from pyomo.contrib.community_detection.event_log import _event_log
def generate_model_graph(model, type_of_graph, with_objective=True, weighted_graph=True, use_only_active_components=True):
    """
    Creates a networkX graph of nodes and edges based on a Pyomo optimization model

    This function takes in a Pyomo optimization model, then creates a graphical representation of the model with
    specific features of the graph determined by the user (see Parameters below).

    (This function is designed to be called by detect_communities, but can be used solely for the purpose of
    creating model graphs as well.)

    Parameters
    ----------
    model: Block
        a Pyomo model or block to be used for community detection
    type_of_graph: str
        a string that specifies the type of graph that is created from the model
        'constraint' creates a graph based on constraint nodes,
        'variable' creates a graph based on variable nodes,
        'bipartite' creates a graph based on constraint and variable nodes (bipartite graph).
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function is included in the graph; the
        default is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether a weighted or unweighted graph is to be created from the Pyomo
        model; the default is True (type_of_graph='bipartite' creates an unweighted graph regardless of this parameter)
    use_only_active_components: bool, optional
        a Boolean argument that specifies whether inactive constraints/objectives are included in the networkX graph

    Returns
    -------
    bipartite_model_graph/projected_model_graph: nx.Graph
        a NetworkX graph with nodes and edges based on the given Pyomo optimization model
    number_component_map: dict
        a dictionary that (deterministically) maps a number to a component in the model
    constraint_variable_map: dict
        a dictionary that maps a numbered constraint to a list of (numbered) variables that appear in the constraint
    """
    edge_set = set()
    bipartite_model_graph = nx.Graph()
    constraint_variable_map = {}
    if with_objective:
        component_number_map = ComponentMap(((component, number) for number, component in enumerate(model.component_data_objects(ctype=(Constraint, Var, Objective), active=use_only_active_components, descend_into=True, sort=SortComponents.deterministic))))
    else:
        component_number_map = ComponentMap(((component, number) for number, component in enumerate(model.component_data_objects(ctype=(Constraint, Var), active=use_only_active_components, descend_into=True, sort=SortComponents.deterministic))))
    number_component_map = dict(((number, comp) for comp, number in component_number_map.items()))
    bipartite_model_graph.add_nodes_from([node_number for node_number in range(len(component_number_map))])
    for model_constraint in model.component_data_objects(ctype=Constraint, active=use_only_active_components, descend_into=True):
        numbered_constraint = component_number_map[model_constraint]
        numbered_variables_in_constraint_equation = [component_number_map[constraint_variable] for constraint_variable in identify_variables(model_constraint.body)]
        constraint_variable_map[numbered_constraint] = numbered_variables_in_constraint_equation
        edges_between_nodes = [(numbered_constraint, numbered_variable_in_constraint) for numbered_variable_in_constraint in numbered_variables_in_constraint_equation]
        edge_set.update(edges_between_nodes)
    if with_objective:
        for objective_function in model.component_data_objects(ctype=Objective, active=use_only_active_components, descend_into=True):
            numbered_objective = component_number_map[objective_function]
            numbered_variables_in_objective = [component_number_map[objective_variable] for objective_variable in identify_variables(objective_function)]
            constraint_variable_map[numbered_objective] = numbered_variables_in_objective
            edges_between_nodes = [(numbered_objective, numbered_variable_in_objective) for numbered_variable_in_objective in numbered_variables_in_objective]
            edge_set.update(edges_between_nodes)
    bipartite_model_graph.add_edges_from(sorted(edge_set))
    if type_of_graph == 'bipartite':
        _event_log(model, bipartite_model_graph, set(constraint_variable_map), type_of_graph, with_objective)
        return (bipartite_model_graph, number_component_map, constraint_variable_map)
    constraint_nodes = set(constraint_variable_map)
    if type_of_graph == 'constraint':
        graph_nodes = constraint_nodes
    else:
        variable_nodes = set(number_component_map) - constraint_nodes
        graph_nodes = variable_nodes
    try:
        if weighted_graph:
            projected_model_graph = nx.bipartite.weighted_projected_graph(bipartite_model_graph, graph_nodes)
        else:
            projected_model_graph = nx.bipartite.projected_graph(bipartite_model_graph, graph_nodes)
    except nx.exception.NetworkXAlgorithmError:
        projected_model_graph = nx.Graph()
    _event_log(model, projected_model_graph, set(constraint_variable_map), type_of_graph, with_objective)
    return (projected_model_graph, number_component_map, constraint_variable_map)