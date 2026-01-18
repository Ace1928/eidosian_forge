import os
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
def new_choose_save_file(title, directory, filename):
    assert directory == str(tmp_path)
    os.makedirs(f'{directory}/test')
    return f'{directory}/test/{filename}'