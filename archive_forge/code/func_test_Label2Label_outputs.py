from ..model import Label2Label
def test_Label2Label_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Label2Label.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value