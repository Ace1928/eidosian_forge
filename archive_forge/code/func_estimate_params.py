import math
import warnings
from .DynamicProgramming import ScaledDPAlgorithms
from Bio import BiopythonDeprecationWarning
def estimate_params(self, transition_counts, emission_counts):
    """Get a maximum likelihood estimation of transition and emmission.

        Arguments:
         - transition_counts -- A dictionary with the total number of counts
           of transitions between two states.
         - emissions_counts -- A dictionary with the total number of counts
           of emmissions of a particular emission letter by a state letter.

        This then returns the maximum likelihood estimators for the
        transitions and emissions, estimated by formulas 3.18 in
        Durbin et al::

            a_{kl} = A_{kl} / sum(A_{kl'})
            e_{k}(b) = E_{k}(b) / sum(E_{k}(b'))

        Returns:
        Transition and emission dictionaries containing the maximum
        likelihood estimators.

        """
    ml_transitions = self.ml_estimator(transition_counts)
    ml_emissions = self.ml_estimator(emission_counts)
    return (ml_transitions, ml_emissions)