import subprocess, os, sys, re, difflib
def failures1_test(suite, *args):
    for results in args:
        assert 'FAILED (failures=2)' in results
        assert 'Ran 18 tests' in results