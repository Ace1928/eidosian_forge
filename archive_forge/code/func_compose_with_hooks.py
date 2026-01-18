import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def compose_with_hooks(self, test_hooks, supports_body=True):
    client = HookMailClient({})
    client.supports_body = supports_body
    for test_hook in test_hooks:
        merge_directive.MergeDirective.hooks.install_named_hook('merge_request_body', test_hook, 'test')
    tree = self.make_branch_and_tree('foo')
    tree.commit('foo')
    directive = merge_directive.MergeDirective2(tree.branch.last_revision(), b'sha', 0, 0, b'sha', source_branch=tree.branch.base, base_revision_id=tree.branch.last_revision(), message='This code rox')
    directive.compose_merge_request(client, 'jrandom@example.com', None, tree.branch)
    return (client, directive)