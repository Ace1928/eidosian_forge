from ..diffusion import TractographyLabelMapSeeding
def test_TractographyLabelMapSeeding_outputs():
    output_map = dict(OutputFibers=dict(extensions=None, position=-1), outputdirectory=dict())
    outputs = TractographyLabelMapSeeding.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value