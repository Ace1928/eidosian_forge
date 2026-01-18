from ..epi import TOPUP
def test_TOPUP_outputs():
    output_map = dict(out_corrected=dict(extensions=None), out_enc_file=dict(extensions=None), out_field=dict(extensions=None), out_fieldcoef=dict(extensions=None), out_jacs=dict(), out_logfile=dict(extensions=None), out_mats=dict(), out_movpar=dict(extensions=None), out_warps=dict())
    outputs = TOPUP.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value