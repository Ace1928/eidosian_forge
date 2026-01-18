from wandb_gql import gql
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
@normalize_exceptions
def sweeps(self):
    query = gql('\n            query GetSweeps($project: String!, $entity: String!) {\n                project(name: $project, entityName: $entity) {\n                    totalSweeps\n                    sweeps {\n                        edges {\n                            node {\n                                ...SweepFragment\n                            }\n                            cursor\n                        }\n                        pageInfo {\n                            endCursor\n                            hasNextPage\n                        }\n                    }\n                }\n            }\n            %s\n            ' % public.SWEEP_FRAGMENT)
    variable_values = {'project': self.name, 'entity': self.entity}
    ret = self.client.execute(query, variable_values)
    if ret['project']['totalSweeps'] < 1:
        return []
    return [public.Sweep(self.client, self.entity, self.name, e['node']['name'], attrs={'id': e['node']['id'], 'name': e['node']['name'], 'bestLoss': e['node']['bestLoss'], 'config': e['node']['config']}) for e in ret['project']['sweeps']['edges']]