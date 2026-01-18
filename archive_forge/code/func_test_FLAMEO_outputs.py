from ..model import FLAMEO
def test_FLAMEO_outputs():
    output_map = dict(copes=dict(), fstats=dict(), mrefvars=dict(), pes=dict(), res4d=dict(), stats_dir=dict(), tdof=dict(), tstats=dict(), var_copes=dict(), weights=dict(), zfstats=dict(), zstats=dict())
    outputs = FLAMEO.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value