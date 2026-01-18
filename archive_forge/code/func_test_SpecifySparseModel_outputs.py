from ..modelgen import SpecifySparseModel
def test_SpecifySparseModel_outputs():
    output_map = dict(session_info=dict(), sparse_png_file=dict(extensions=None), sparse_svg_file=dict(extensions=None))
    outputs = SpecifySparseModel.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value