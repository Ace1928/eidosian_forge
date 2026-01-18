from ..utilities import GeneratePurePlugMask
def test_GeneratePurePlugMask_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputImageModalities=dict(argstr='--inputImageModalities %s...'), numberOfSubSamples=dict(argstr='--numberOfSubSamples %s', sep=','), outputMaskFile=dict(argstr='--outputMaskFile %s', hash_files=False), threshold=dict(argstr='--threshold %f'))
    inputs = GeneratePurePlugMask.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value