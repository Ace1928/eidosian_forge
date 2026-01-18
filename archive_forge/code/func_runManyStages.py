import time
import pyqtgraph as pg
def runManyStages(i):
    """Iterate over runStage() 3 times while incrementing a progress bar.
    """
    with pg.ProgressDialog('Running stage %s..' % i, maximum=3, nested=True, wait=0) as dlg:
        for j in range(1, 4):
            runStage('%d.%d' % (i, j))
            dlg += 1
            if dlg.wasCanceled():
                print('Canceled stage %s' % i)
                break