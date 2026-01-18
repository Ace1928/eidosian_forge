from time import sleep
import pytest
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
def bad_func():
    raise Exception