import subprocess, os, sys, re, difflib
def assert_on_results(suite, single, sub):
    test = globals().get(f'{suite}_test')
    if hasattr(test, '__call_'):
        test(suite, single, sub)
        print(f'assertions on {suite} OK')