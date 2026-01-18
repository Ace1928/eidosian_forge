import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List
from typing import List, Tuple
from functools import wraps
import logging
@StandardDecorator()
def calculate_smoothness_and_monotonicity(board: np.ndarray) -> Tuple[float, float]:
    smoothness = 0
    monotonicity_up_down = 0
    monotonicity_left_right = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1] - 1):
            if board[i, j] != 0 and board[i, j + 1] != 0:
                smoothness -= abs(np.log2(board[i, j]) - np.log2(board[i, j + 1]))
            if board[j, i] != 0 and board[j + 1, i] != 0:
                smoothness -= abs(np.log2(board[j, i]) - np.log2(board[j + 1, i]))
    for i in range(board.shape[0]):
        for j in range(1, board.shape[1]):
            if board[i, j - 1] > board[i, j]:
                monotonicity_left_right += np.log2(board[i, j - 1]) - np.log2(board[i, j])
            else:
                monotonicity_left_right -= np.log2(board[i, j]) - np.log2(board[i, j - 1])
            if board[j - 1, i] > board[j, i]:
                monotonicity_up_down += np.log2(board[j - 1, i]) - np.log2(board[j, i])
            else:
                monotonicity_up_down -= np.log2(board[j, i]) - np.log2(board[j - 1, i])
    return (smoothness, (monotonicity_left_right + monotonicity_up_down) / 2)