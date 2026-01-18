from ..preprocess import Tensor2FractionalAnisotropy
def test_Tensor2FractionalAnisotropy_outputs():
    output_map = dict(FA=dict(extensions=None))
    outputs = Tensor2FractionalAnisotropy.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value