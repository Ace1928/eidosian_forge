from ..model import Randomise
def test_Randomise_outputs():
    output_map = dict(f_corrected_p_files=dict(), f_p_files=dict(), fstat_files=dict(), t_corrected_p_files=dict(), t_p_files=dict(), tstat_files=dict())
    outputs = Randomise.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value