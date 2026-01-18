import pathlib
import shutil
import string
from tempfile import mkdtemp
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from cirq.devices import grid_qubit
from cirq.vis import heatmap
@pytest.mark.parametrize('format_string', ['.3e', '.2f', '.4g'])
def test_annotation_position_and_content(ax, format_string):
    row_col_list = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    qubits = [grid_qubit.GridQubit(row, col) for row, col in row_col_list]
    values = np.random.random(len(qubits))
    test_value_map = {(qubit,): value for qubit, value in zip(qubits, values)}
    test_row_col_map = {rc: value for rc, value in zip(row_col_list, values)}
    random_heatmap = heatmap.Heatmap(test_value_map, annotation_format=format_string)
    random_heatmap.plot(ax)
    actual_texts = set()
    for artist in ax.get_children():
        if isinstance(artist, mpl.text.Text):
            col, row = artist.get_position()
            text = artist.get_text()
            actual_texts.add(((row, col), text))
    expected_texts = set(((qubit, format(value, format_string)) for qubit, value in test_row_col_map.items()))
    assert expected_texts.issubset(actual_texts)