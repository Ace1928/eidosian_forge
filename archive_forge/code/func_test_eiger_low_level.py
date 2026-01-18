from ..common import ut
import numpy as np
import h5py as h5
import tempfile
def test_eiger_low_level(self):
    self.outfile = self.working_dir + 'eiger.h5'
    with h5.File(self.outfile, 'w', libver='latest') as f:
        vdset_shape = (78, 200, 200)
        vdset_max_shape = vdset_shape
        virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)
        dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
        dcpl.set_fill_value(np.array([-1]))
        k = 0
        for foo in self.fname:
            in_data = h5.File(foo, 'r')['data']
            src_shape = in_data.shape
            max_src_shape = src_shape
            in_data.file.close()
            src_dspace = h5.h5s.create_simple(src_shape, max_src_shape)
            src_dspace.select_hyperslab(start=(0, 0, 0), stride=(1, 1, 1), count=(1, 1, 1), block=src_shape)
            virt_dspace.select_hyperslab(start=(k, 0, 0), stride=(1, 1, 1), count=(1, 1, 1), block=src_shape)
            dcpl.set_virtual(virt_dspace, foo.encode('utf-8'), b'data', src_dspace)
            k += src_shape[0]
        h5.h5d.create(f.id, name=b'data', tid=h5.h5t.NATIVE_INT16, space=virt_dspace, dcpl=dcpl)
    f = h5.File(self.outfile, 'r')['data']
    self.assertEqual(f[10, 100, 10], 0.0)
    self.assertEqual(f[30, 100, 100], 1.0)
    self.assertEqual(f[50, 100, 100], 2.0)
    self.assertEqual(f[70, 100, 100], 3.0)
    f.file.close()