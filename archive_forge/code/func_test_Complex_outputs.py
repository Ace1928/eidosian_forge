from ..utils import Complex
def test_Complex_outputs():
    output_map = dict(complex_out_file=dict(extensions=None), imaginary_out_file=dict(extensions=None), magnitude_out_file=dict(extensions=None), phase_out_file=dict(extensions=None), real_out_file=dict(extensions=None))
    outputs = Complex.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value