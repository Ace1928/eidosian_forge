from parlai.core.worlds import World
@staticmethod
def generate_world(opt, agents):
    return OnboardWorld(opt, agents[0])