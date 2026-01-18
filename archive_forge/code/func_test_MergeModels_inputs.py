from ..surface import MergeModels
def test_MergeModels_inputs():
    input_map = dict(Model1=dict(argstr='%s', extensions=None, position=-3), Model2=dict(argstr='%s', extensions=None, position=-2), ModelOutput=dict(argstr='%s', hash_files=False, position=-1), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True))
    inputs = MergeModels.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value