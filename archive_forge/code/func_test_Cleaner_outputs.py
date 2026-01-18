from ..fix import Cleaner
def test_Cleaner_outputs():
    output_map = dict(cleaned_functional_file=dict(extensions=None))
    outputs = Cleaner.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value