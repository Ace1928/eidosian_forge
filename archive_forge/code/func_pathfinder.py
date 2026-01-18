import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from multiprocessing import cpu_count
from typing import (
import pandas as pd
from tqdm.auto import tqdm
from cmdstanpy import (
from cmdstanpy.cmdstan_args import (
from cmdstanpy.stanfit import (
from cmdstanpy.utils import (
from cmdstanpy.utils.filesystem import temp_inits, temp_single_json
from . import progress as progbar
def pathfinder(self, data: Union[Mapping[str, Any], str, os.PathLike, None]=None, *, init_alpha: Optional[float]=None, tol_obj: Optional[float]=None, tol_rel_obj: Optional[float]=None, tol_grad: Optional[float]=None, tol_rel_grad: Optional[float]=None, tol_param: Optional[float]=None, history_size: Optional[int]=None, num_paths: Optional[int]=None, max_lbfgs_iters: Optional[int]=None, draws: Optional[int]=None, num_single_draws: Optional[int]=None, num_elbo_draws: Optional[int]=None, psis_resample: bool=True, calculate_lp: bool=True, seed: Optional[int]=None, inits: Union[Dict[str, float], float, str, os.PathLike, None]=None, output_dir: OptionalPath=None, sig_figs: Optional[int]=None, save_profile: bool=False, show_console: bool=False, refresh: Optional[int]=None, time_fmt: str='%Y%m%d%H%M%S', timeout: Optional[float]=None, num_threads: Optional[int]=None) -> CmdStanPathfinder:
    """
        Run CmdStan's Pathfinder variational inference algorithm.

        :param data: Values for all data variables in the model, specified
            either as a dictionary with entries matching the data variables,
            or as the path of a data file in JSON or Rdump format.

        :param num_paths: Number of single-path Pathfinders to run.
            Default is 4, when the number of paths is 1 then no importance
            sampling is done.

        :param draws: Number of approximate draws to return.

        :param num_single_draws: Number of draws each single-pathfinder will
            draw.
            If ``num_paths`` is 1, only one of this and ``draws`` should be
            used.

        :param max_lbfgs_iters: Maximum number of L-BFGS iterations.

        :param num_elbo_draws: Number of Monte Carlo draws to evaluate ELBO.

        :param psis_resample: Whether or not to use Pareto Smoothed Importance
            Sampling on the result of the individual Pathfinders. If False, the
            result contains the draws from each path.

        :param calculate_lp: Whether or not to calculate the log probability
            for approximate draws. If False, this also implies that
            ``psis_resample`` will be set to False.

        :param seed: The seed for random number generator. Must be an integer
            between 0 and 2^32 - 1. If unspecified,
            :func:`numpy.random.default_rng` is used to generate a seed.

        :param inits: Specifies how the algorithm initializes parameter values.
            Initialization is either uniform random on a range centered on 0,
            exactly 0, or a dictionary or file of initial values for some or all
            parameters in the model.  The default initialization behavior will
            initialize all parameter values on range [-2, 2] on the
            *unconstrained* support.  If the expected parameter values are
            too far from this range, this option may improve adaptation.
            The following value types are allowed:

            * Single number n > 0 - initialization range is [-n, n].
            * 0 - all parameters are initialized to 0.
            * dictionary - pairs parameter name : initial value.
            * string - pathname to a JSON or Rdump data file.
            * list of strings - per-path pathname to data file.
            * list of dictionaries - per-path initial values.

        :param init_alpha: For internal L-BFGS: Line search step size for
            first iteration

        :param tol_obj: For internal L-BFGS: Convergence tolerance on changes
            in objective function value

        :param tol_rel_obj: For internal L-BFGS: Convergence tolerance on
            relative changes in objective function value

        :param tol_grad: For internal L-BFGS: Convergence tolerance on the
            norm of the gradient

        :param tol_rel_grad: For internal L-BFGS: Convergence tolerance on
            the relative norm of the gradient

        :param tol_param: For internal L-BFGS: Convergence tolerance on changes
            in parameter value

        :param history_size: For internal L-BFGS: Size of the history for LBFGS
            Hessian approximation. The value should be less than the
            dimensionality of the parameter space. 5-10 is usually sufficient

        :param output_dir: Name of the directory to which CmdStan output
            files are written. If unspecified, output files will be written
            to a temporary directory which is deleted upon session exit.

        :param sig_figs: Numerical precision used for output CSV and text files.
            Must be an integer between 1 and 18.  If unspecified, the default
            precision for the system file I/O is used; the usual value is 6.
            Introduced in CmdStan-2.25.

        :param save_profile: Whether or not to profile auto-diff operations in
            labelled blocks of code.  If ``True``, CSV outputs are written to
            file '<model_name>-<YYYYMMDDHHMM>-profile-<path_id>'.
            Introduced in CmdStan-2.26, see
            https://mc-stan.org/docs/cmdstan-guide/stan_csv.html,
            section "Profiling CSV output file" for details.

        :param show_console: If ``True``, stream CmdStan messages sent to stdout
            and stderr to the console.  Default is ``False``.

        :param refresh: Specify the number of iterations CmdStan will take
            between progress messages. Default value is 100.

        :param time_fmt: A format string passed to
            :meth:`~datetime.datetime.strftime` to decide the file names for
            output CSVs. Defaults to "%Y%m%d%H%M%S"

        :param timeout: Duration at which Pathfinder times
            out in seconds. Defaults to None.

        :param num_threads: Number of threads to request for parallel execution.
            A number other than ``1`` requires the model to have been compiled
            with STAN_THREADS=True.

        :return: A :class:`CmdStanPathfinder` object

        References
        ----------

        Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder:
        Parallel quasi-Newton variational inference. Journal of Machine Learning
        Research, 23(306), 1–49. Retrieved from
        http://jmlr.org/papers/v23/21-0889.html
        """
    exe_info = self.exe_info()
    if cmdstan_version_before(2, 33, exe_info):
        raise ValueError("Method 'pathfinder' not available for CmdStan versions before 2.33")
    if (not psis_resample or not calculate_lp) and cmdstan_version_before(2, 34, exe_info):
        raise ValueError("Arguments 'psis_resample' and 'calculate_lp' are only available for CmdStan versions 2.34 and later")
    if num_threads is not None:
        if num_threads != 1 and exe_info.get('STAN_THREADS', '').lower() != 'true':
            raise ValueError("Model must be compiled with 'STAN_THREADS=true' to use 'num_threads' argument")
        os.environ['STAN_NUM_THREADS'] = str(num_threads)
    if num_paths == 1:
        if num_single_draws is None:
            num_single_draws = draws
        if draws is not None and num_single_draws != draws:
            raise ValueError("Cannot specify both 'draws' and 'num_single_draws' when 'num_paths' is 1")
    pathfinder_args = PathfinderArgs(init_alpha=init_alpha, tol_obj=tol_obj, tol_rel_obj=tol_rel_obj, tol_grad=tol_grad, tol_rel_grad=tol_rel_grad, tol_param=tol_param, history_size=history_size, num_psis_draws=draws, num_paths=num_paths, max_lbfgs_iters=max_lbfgs_iters, num_draws=num_single_draws, num_elbo_draws=num_elbo_draws, psis_resample=psis_resample, calculate_lp=calculate_lp)
    with temp_single_json(data) as _data, temp_inits(inits) as _inits:
        args = CmdStanArgs(self._name, self._exe_file, chain_ids=None, data=_data, seed=seed, inits=_inits, output_dir=output_dir, sig_figs=sig_figs, save_profile=save_profile, method_args=pathfinder_args, refresh=refresh)
        dummy_chain_id = 0
        runset = RunSet(args=args, chains=1, time_fmt=time_fmt)
        self._run_cmdstan(runset, dummy_chain_id, show_console=show_console, timeout=timeout)
    runset.raise_for_timeouts()
    if not runset._check_retcodes():
        msg = "Error during Pathfinder! Command '{}' failed: {}".format(' '.join(runset.cmd(0)), runset.get_err_msgs())
        raise RuntimeError(msg)
    return CmdStanPathfinder(runset)