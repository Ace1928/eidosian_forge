from ..nx import NetworkXMetrics
def test_NetworkXMetrics_inputs():
    input_map = dict(compute_clique_related_measures=dict(usedefault=True), in_file=dict(extensions=None, mandatory=True), out_edge_metrics_matlab=dict(extensions=None, genfile=True), out_global_metrics_matlab=dict(extensions=None, genfile=True), out_k_core=dict(extensions=None, usedefault=True), out_k_crust=dict(extensions=None, usedefault=True), out_k_shell=dict(extensions=None, usedefault=True), out_node_metrics_matlab=dict(extensions=None, genfile=True), out_pickled_extra_measures=dict(extensions=None, usedefault=True), treat_as_weighted_graph=dict(usedefault=True))
    inputs = NetworkXMetrics.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value