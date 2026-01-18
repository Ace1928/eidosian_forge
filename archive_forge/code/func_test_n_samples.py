from datetime import datetime
from time import sleep
from pytest import raises
from tune.concepts.flow.report import TrialReport
from tune.concepts.flow.trial import Trial
from tune.noniterative.stopper import (
def test_n_samples():
    r1 = mr([], 0.1)
    r2 = mr([], 0.4)
    r3 = mr([], 0.3)
    s = n_samples(2)
    assert s.can_accept(r1.trial)
    s.judge(r1)
    assert s.can_accept(r2.trial)
    s.judge(r2)
    assert not s.can_accept(r3.trial)