import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def confidence_region_test(self, theta_values, distribution, alphas, test_theta_values=None):
    """
        Confidence region test to determine if theta values are within a
        rectangular, multivariate normal, or Gaussian kernel density distribution
        for a range of alpha values

        Parameters
        ----------
        theta_values: pd.DataFrame, columns = theta_names
            Theta values used to generate a confidence region
            (generally returned by theta_est_bootstrap)
        distribution: string
            Statistical distribution used to define a confidence region,
            options = 'MVN' for multivariate_normal, 'KDE' for gaussian_kde,
            and 'Rect' for rectangular.
        alphas: list
            List of alpha values used to determine if theta values are inside
            or outside the region.
        test_theta_values: pd.Series or pd.DataFrame, keys/columns = theta_names, optional
            Additional theta values that are compared to the confidence region
            to determine if they are inside or outside.

        Returns
        training_results: pd.DataFrame
            Theta value used to generate the confidence region along with True
            (inside) or False (outside) for each alpha
        test_results: pd.DataFrame
            If test_theta_values is not None, returns test theta value along
            with True (inside) or False (outside) for each alpha
        """
    assert isinstance(theta_values, pd.DataFrame)
    assert distribution in ['Rect', 'MVN', 'KDE']
    assert isinstance(alphas, list)
    assert isinstance(test_theta_values, (type(None), dict, pd.Series, pd.DataFrame))
    if isinstance(test_theta_values, (dict, pd.Series)):
        test_theta_values = pd.Series(test_theta_values).to_frame().transpose()
    training_results = theta_values.copy()
    if test_theta_values is not None:
        test_result = test_theta_values.copy()
    for a in alphas:
        if distribution == 'Rect':
            lb, ub = graphics.fit_rect_dist(theta_values, a)
            training_results[a] = (theta_values > lb).all(axis=1) & (theta_values < ub).all(axis=1)
            if test_theta_values is not None:
                test_result[a] = (test_theta_values > lb).all(axis=1) & (test_theta_values < ub).all(axis=1)
        elif distribution == 'MVN':
            dist = graphics.fit_mvn_dist(theta_values)
            Z = dist.pdf(theta_values)
            score = scipy.stats.scoreatpercentile(Z, (1 - a) * 100)
            training_results[a] = Z >= score
            if test_theta_values is not None:
                Z = dist.pdf(test_theta_values)
                test_result[a] = Z >= score
        elif distribution == 'KDE':
            dist = graphics.fit_kde_dist(theta_values)
            Z = dist.pdf(theta_values.transpose())
            score = scipy.stats.scoreatpercentile(Z, (1 - a) * 100)
            training_results[a] = Z >= score
            if test_theta_values is not None:
                Z = dist.pdf(test_theta_values.transpose())
                test_result[a] = Z >= score
    if test_theta_values is not None:
        return (training_results, test_result)
    else:
        return training_results