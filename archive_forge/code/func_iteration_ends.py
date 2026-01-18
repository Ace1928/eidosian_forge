import numpy as np
def iteration_ends(self, time_step):
    """Perform updates to learning rate and potential other states at the
        end of an iteration

        Parameters
        ----------
        time_step : int
            number of training samples trained on so far, used to update
            learning rate for 'invscaling'
        """
    if self.lr_schedule == 'invscaling':
        self.learning_rate = float(self.learning_rate_init) / (time_step + 1) ** self.power_t