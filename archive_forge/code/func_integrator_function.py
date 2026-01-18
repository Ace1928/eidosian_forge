import networkx as nx
def integrator_function(inputs):
    aggregated_value = sum(inputs)
    threshold = 3
    return 1 if aggregated_value > threshold else 0