import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch.optim as optim
import os
from ActivationDictionary import ActivationDictionary
from IndegoLogging import configure_logging
def calculate_reward(current_loss: float, previous_loss: float, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the reward based on the improvement in loss and the combined metrics of precision,
    recall, and F1 score. This function ensures that the true and predicted labels have the same length,
    computes the improvement in loss, and calculates the precision, recall, and F1 score to determine
    the reward.

    Args:
    current_loss (float): The current loss after a model update.
    previous_loss (float): The loss before the model update.
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted labels by the model.

    Returns:
    float: The calculated reward based on the specified metrics.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        logger.error('The true labels and predicted labels arrays do not match in length.')
        raise ValueError('The length of true labels and predicted labels must be the same.')
    loss_improvement: float = previous_loss - current_loss
    logger.debug(f'Calculated loss improvement: {loss_improvement}')
    metrics: Dict[str, float] = {'precision': float(precision_score(y_true, y_pred, average='macro')), 'recall': float(recall_score(y_true, y_pred, average='macro')), 'f1': float(f1_score(y_true, y_pred, average='macro'))}
    logger.info(f'Reward calculation details: Loss Improvement: {loss_improvement}, Metrics: {metrics}')
    reward: float = loss_improvement + 0.5 * sum(metrics.values())
    logger.debug(f'Constructed reward using weighted metrics: {reward}')
    return reward