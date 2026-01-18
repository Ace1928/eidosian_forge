from ..csv import CSVReader
def test_CSVReader_outputs():
    output_map = dict()
    outputs = CSVReader.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value