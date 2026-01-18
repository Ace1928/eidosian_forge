from ..nx import AverageNetworks
def test_AverageNetworks_inputs():
    input_map = dict(group_id=dict(usedefault=True), in_files=dict(mandatory=True), out_gexf_groupavg=dict(extensions=None), out_gpickled_groupavg=dict(extensions=None), resolution_network_file=dict(extensions=None))
    inputs = AverageNetworks.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value