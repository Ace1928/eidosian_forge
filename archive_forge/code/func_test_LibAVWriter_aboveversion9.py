import os
import sys
import numpy as np
from numpy.testing import assert_equal, assert_array_less
import skvideo
import skvideo.datasets
import skvideo.io
@unittest.skipIf(not skvideo._HAS_AVCONV, 'LibAV required for this test.')
def test_LibAVWriter_aboveversion9():
    if not skvideo._HAS_AVCONV:
        return 0
    if np.int(skvideo._LIBAV_MAJOR_VERSION) < 9:
        return 0
    outputfile = sys._getframe().f_code.co_name + '.mp4'
    outputdata = np.random.random(size=(5, 480, 640, 3)) * 255
    outputdata = outputdata.astype(np.uint8)
    writer = skvideo.io.LibAVWriter(outputfile, verbosity=0)
    for i in range(5):
        writer.writeFrame(outputdata[i])
    writer.close()
    os.remove(outputfile)