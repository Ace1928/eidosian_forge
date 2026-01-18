import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
def change_type(self, new_type: str) -> None:
    """Change the type of the artifact collection.

        Arguments:
            new_type: The new collection type to use, freeform string.
        """
    if not self.is_sequence():
        raise ValueError('Artifact collection needs to be a sequence')
    termlog(f'Changing artifact collection type of {self.type} to {new_type}')
    template = '\n            mutation MoveArtifactCollection(\n                $artifactSequenceID: ID!\n                $destinationArtifactTypeName: String!\n            ) {\n                moveArtifactSequence(\n                input: {\n                    artifactSequenceID: $artifactSequenceID\n                    destinationArtifactTypeName: $destinationArtifactTypeName\n                }\n                ) {\n                artifactCollection {\n                    id\n                    name\n                    description\n                    __typename\n                }\n                }\n            }\n            '
    variable_values = {'artifactSequenceID': self.id, 'destinationArtifactTypeName': new_type}
    mutation = gql(template)
    self.client.execute(mutation, variable_values=variable_values)
    self.type = new_type