import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pyquil.experiment._setting import ExperimentSetting
def correct_experiment_result(result: ExperimentResult, calibration: ExperimentResult) -> ExperimentResult:
    """
    Given a raw, unmitigated result and its associated readout calibration, produce the result
    absent readout error.

    :param result: An ``ExperimentResult`` object with unmitigated readout error.
    :param calibration: An ``ExperimentResult`` object resulting from running readout calibration
        on the ``ExperimentSetting`` associated with the ``result`` parameter.
    :return: An ``ExperimentResult`` object corrected for symmetric readout error.
    """
    corrected_expectation = result.expectation / calibration.expectation
    assert result.std_err is not None and calibration.std_err is not None
    corrected_variance = ratio_variance(result.expectation, result.std_err ** 2, calibration.expectation, calibration.std_err ** 2)
    additional_results = None
    if result.additional_results is not None and calibration.additional_results:
        assert len(result.additional_results) == len(calibration.additional_results)
        additional_results = [correct_experiment_result(r, c) for r, c in zip(result.additional_results, calibration.additional_results)]
    return ExperimentResult(setting=result.setting, expectation=corrected_expectation, std_err=np.sqrt(corrected_variance).item(), total_counts=result.total_counts, raw_expectation=result.expectation, raw_std_err=result.std_err, calibration_expectation=calibration.expectation, calibration_std_err=calibration.std_err, calibration_counts=calibration.total_counts, additional_results=additional_results)