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
def expectimax(board: np.ndarray, depth: int, playerTurn: bool) -> Tuple[float, str]:
    """
    Performs the expectimax search to evaluate the best move for the current game state by exploring all possible moves
    and their outcomes up to a specified depth. This function alternates between maximizing the player's score and
    evaluating the expected value of chance nodes, thereby simulating the game's stochastic nature.

    Args:
        board (np.ndarray): The current game board represented as a 2D NumPy array.
        depth (int): The depth of the search, indicating how many moves ahead the algorithm should evaluate.
        playerTurn (bool): A boolean flag indicating whether it's the player's turn to move or a chance node.

    Returns:
        Tuple[float, str]: A tuple containing the best heuristic value found for the current player's turn and the
                            corresponding move as a string. If it's a chance node, the move will be an empty string.
    """
    logging.debug(f'Initiating expectimax with depth {depth} and playerTurn {playerTurn}.')
    if depth == 0 or is_game_over(board):
        heuristic = heuristic_evaluation(board)
        logging.info(f'Reached base case with heuristic value: {heuristic}.')
        return (heuristic, '')
    if playerTurn:
        best_value, best_move = (float('-inf'), '')
        for move in ['left', 'right', 'up', 'down']:
            new_board, _, _ = calculate_best_move(board, move)
            value, _ = expectimax(new_board, depth - 1, False)
            if value > best_value:
                best_value, best_move = (value, move)
            logging.debug(f'Evaluating player move: {move} with value: {value}.')
        logging.info(f'Best move for player: {best_move} with value: {best_value}.')
        return (best_value, best_move)
    else:
        total_value, empty_tiles = (0, get_empty_tiles(board))
        for i, j in empty_tiles:
            for value in [2, 4]:
                new_board = np.array(board)
                new_board[i, j] = value
                value, _ = expectimax(new_board, depth - 1, True)
                probability = 0.9 if value == 2 else 0.1
                total_value += probability * value / len(empty_tiles)
                logging.debug(f'Chance node with tile {value} at ({i},{j}) evaluated with value: {value}.')
        logging.info(f'Total value for chance node: {total_value}.')
        return (total_value, '')