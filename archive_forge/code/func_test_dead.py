import time
from IPython.lib import backgroundjobs as bg
def test_dead():
    """Test control of dead jobs"""
    jobs = bg.BackgroundJobManager()
    j = jobs.new(crasher)
    j.join()
    assert len(jobs.completed) == 0
    assert len(jobs.dead) == 1
    jobs.flush()
    assert len(jobs.dead) == 0