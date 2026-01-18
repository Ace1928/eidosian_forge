import time
from IPython.lib import backgroundjobs as bg
def test_longer():
    """Test control of longer-running jobs"""
    jobs = bg.BackgroundJobManager()
    j = jobs.new(sleeper, 0.1)
    assert len(jobs.running) == 1
    assert len(jobs.completed) == 0
    j.join()
    assert len(jobs.running) == 0
    assert len(jobs.completed) == 1