from ..diffusion import DWIConvert
def test_DWIConvert_outputs():
    output_map = dict(gradientVectorFile=dict(extensions=None), outputBValues=dict(extensions=None), outputBVectors=dict(extensions=None), outputDirectory=dict(), outputVolume=dict(extensions=None))
    outputs = DWIConvert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value