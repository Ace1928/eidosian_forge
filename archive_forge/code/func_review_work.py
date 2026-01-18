from parlai.core.worlds import World
from parlai.mturk.core.dev.agents import AssignState
def review_work(self):
    """
        Programmatically approve/reject the turker's work. Doing this now (if possible)
        means that you don't need to do the work of reviewing later on.

        For example:
        .. code-block:: python
            if self.turker_response == '0':
                self.mturk_agent.reject_work(
                    'You rated our model's response as a 0/10 but we '
                    'know we're better than that'
                )
            else:
                if self.turker_response == '10':
                    self.mturk_agent.pay_bonus(1, 'Thanks for a great rating!')
                self.mturk_agent.approve_work()
        """
    pass