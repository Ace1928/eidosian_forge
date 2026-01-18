import gym
@property
def render_mode(self):
    """Returns the collection render_mode name."""
    return f'{self.env.render_mode}_list'