from lognormal_around import lognormal_around
import numpy as np
import random
def process_input_with_subthreshold_oscillation(self, input_signal, time_step, time_scale=0.01):
    """
        Processes input signal with subthreshold oscillation and stochastic firing.
        """
    oscillation = self.oscillation_amplitude * np.sin(2 * np.pi * self.oscillation_base_frequency * time_step * time_scale)
    modulated_input = input_signal + oscillation
    firing_probability = self.calculate_firing_probability(modulated_input)
    if random.random() < firing_probability:
        self.output = self.base_signal_strength
        self.feedback_signal = (modulated_input - self.threshold) / self.threshold
        self.update_refractory_period(self.feedback_signal)
        self.update_threshold()
        self.firing_history.append(time_step)
    else:
        self.output = 0
        self.feedback_signal = 0