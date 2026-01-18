from ..calib import SFPICOCalibData
def test_SFPICOCalibData_outputs():
    output_map = dict(PICOCalib=dict(extensions=None), calib_info=dict(extensions=None))
    outputs = SFPICOCalibData.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value