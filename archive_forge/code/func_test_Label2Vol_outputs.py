from ..model import Label2Vol
def test_Label2Vol_outputs():
    output_map = dict(vol_label_file=dict(extensions=None))
    outputs = Label2Vol.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value