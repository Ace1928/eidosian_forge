from ..model import Label2Annot
def test_Label2Annot_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Label2Annot.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value