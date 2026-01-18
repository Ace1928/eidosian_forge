from ..io import DataSink
def test_DataSink_inputs():
    input_map = dict(_outputs=dict(usedefault=True), base_directory=dict(), bucket=dict(), container=dict(), creds_path=dict(), encrypt_bucket_keys=dict(), local_copy=dict(), parameterization=dict(usedefault=True), regexp_substitutions=dict(), remove_dest_dir=dict(usedefault=True), strip_dir=dict(), substitutions=dict())
    inputs = DataSink.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value