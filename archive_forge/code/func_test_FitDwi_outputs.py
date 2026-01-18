from ..dwi import FitDwi
def test_FitDwi_outputs():
    output_map = dict(error_file=dict(extensions=None), famap_file=dict(extensions=None), mcmap_file=dict(extensions=None), mcout=dict(extensions=None), mdmap_file=dict(extensions=None), nodiff_file=dict(extensions=None), res_file=dict(extensions=None), rgbmap_file=dict(extensions=None), syn_file=dict(extensions=None), tenmap2_file=dict(extensions=None), tenmap_file=dict(extensions=None), v1map_file=dict(extensions=None))
    outputs = FitDwi.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value