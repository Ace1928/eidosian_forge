import warnings
def confirm_action(self, prompt, confirmation_id, args):
    return self.get_boolean(prompt % args)