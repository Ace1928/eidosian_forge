from ..utils import ImageInfo
def test_ImageInfo_outputs():
    output_map = dict(TE=dict(), TI=dict(), TR=dict(), data_type=dict(), dimensions=dict(), file_format=dict(), info=dict(), orientation=dict(), out_file=dict(extensions=None), ph_enc_dir=dict(), vox_sizes=dict())
    outputs = ImageInfo.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value