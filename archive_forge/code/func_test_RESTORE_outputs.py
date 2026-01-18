from ..reconstruction import RESTORE
def test_RESTORE_outputs():
    output_map = dict(evals=dict(extensions=None), evecs=dict(extensions=None), fa=dict(extensions=None), md=dict(extensions=None), mode=dict(extensions=None), rd=dict(extensions=None), trace=dict(extensions=None))
    outputs = RESTORE.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value