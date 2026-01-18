from ..nbs import NetworkBasedStatistic
def test_NetworkBasedStatistic_inputs():
    input_map = dict(edge_key=dict(usedefault=True), in_group1=dict(mandatory=True), in_group2=dict(mandatory=True), node_position_network=dict(extensions=None), number_of_permutations=dict(usedefault=True), out_nbs_network=dict(extensions=None), out_nbs_pval_network=dict(extensions=None), t_tail=dict(usedefault=True), threshold=dict(usedefault=True))
    inputs = NetworkBasedStatistic.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value