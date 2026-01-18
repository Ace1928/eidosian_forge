import gc
import itertools as it
from timeit import timeit
from unittest import mock
import numpy as np
import nibabel as nib
from nibabel.openers import HAVE_INDEXED_GZIP
from nibabel.tmpdirs import InTemporaryDirectory
from ..rstutils import rst_table
from .butils import print_git_title
def bench_arrayproxy_slicing():
    print_git_title('\nArrayProxy gzip slicing')
    tests = list(it.product(HAVE_IGZIP, KEEP_OPENS, SLICEOBJS))
    tests = [t for t in tests if not (t[0] and (not t[1]))]
    testfile = 'testfile.nii'
    testfilegz = 'test.nii.gz'

    def get_test_label(test):
        have_igzip = test[0]
        keep_open = test[1]
        if not (have_igzip and keep_open):
            return 'gzip'
        else:
            return 'indexed_gzip'

    def fix_sliceobj(sliceobj):
        new_sliceobj = []
        for i, s in enumerate(sliceobj):
            if s == ':':
                new_sliceobj.append(slice(None))
            elif s == '?':
                new_sliceobj.append(np.random.randint(0, SHAPE[i]))
            else:
                new_sliceobj.append(int(s * SHAPE[i]))
        return tuple(new_sliceobj)

    def fmt_sliceobj(sliceobj):
        slcstr = []
        for i, s in enumerate(sliceobj):
            if s in ':?':
                slcstr.append(s)
            else:
                slcstr.append(str(int(s * SHAPE[i])))
        return f'[{', '.join(slcstr)}]'
    with InTemporaryDirectory():
        print(f'Generating test data... ({int(round(np.prod(SHAPE) * 4 / 1048576.0))} MB)')
        data = np.array(np.random.random(SHAPE), dtype=np.float32)
        mask = np.random.random(SHAPE[:3]) > 0.1
        if len(SHAPE) > 3:
            data[mask, :] = 0
        else:
            data[mask] = 0
        img = nib.nifti1.Nifti1Image(data, np.eye(4))
        nib.save(img, testfilegz)
        nib.save(img, testfile)
        results = []
        seeds = [np.random.randint(0, 2 ** 32) for s in SLICEOBJS]
        for ti, test in enumerate(tests):
            label = get_test_label(test)
            have_igzip, keep_open, sliceobj = test
            seed = seeds[SLICEOBJS.index(sliceobj)]
            print(f'Running test {ti + 1} of {len(tests)} ({label})...')
            img = nib.load(testfile, keep_file_open=keep_open)
            with mock.patch('nibabel.openers.HAVE_INDEXED_GZIP', have_igzip):
                imggz = nib.load(testfilegz, keep_file_open=keep_open)

            def basefunc():
                img.dataobj[fix_sliceobj(sliceobj)]

            def testfunc():
                with mock.patch('nibabel.openers.HAVE_INDEXED_GZIP', have_igzip):
                    imggz.dataobj[fix_sliceobj(sliceobj)]
            gc.collect()
            if memory_usage is not None:
                membaseline = max(memory_usage(lambda: None))
                testmem = max(memory_usage(testfunc)) - membaseline
                basemem = max(memory_usage(basefunc)) - membaseline
            else:
                testmem = np.nan
                basemem = np.nan
            np.random.seed(seed)
            testtime = float(timeit(testfunc, number=NITERS)) / float(NITERS)
            np.random.seed(seed)
            basetime = float(timeit(basefunc, number=NITERS)) / float(NITERS)
            results.append((label, keep_open, sliceobj, testtime, basetime, testmem, basemem))
    data = np.zeros((len(results), 4))
    data[:, 0] = [r[3] for r in results]
    data[:, 1] = [r[4] for r in results]
    try:
        data[:, 2] = [r[3] / r[4] for r in results]
    except ZeroDivisionError:
        data[:, 2] = np.nan
    data[:, 3] = [r[5] - r[6] for r in results]
    rowlbls = [f'Type {r[0]}, keep_open {r[1]}, slice {fmt_sliceobj(r[2])}' for r in results]
    collbls = ['Time', 'Baseline time', 'Time ratio', 'Memory deviation']
    print(rst_table(data, rowlbls, collbls))