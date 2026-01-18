from ..dti import ProbTrackX2
def test_ProbTrackX2_outputs():
    output_map = dict(fdt_paths=dict(), log=dict(extensions=None), lookup_tractspace=dict(extensions=None), matrix1_dot=dict(extensions=None), matrix2_dot=dict(extensions=None), matrix3_dot=dict(extensions=None), network_matrix=dict(extensions=None), particle_files=dict(), targets=dict(), way_total=dict(extensions=None))
    outputs = ProbTrackX2.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value