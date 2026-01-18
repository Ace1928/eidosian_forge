from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from langchain_core._api import deprecated
from langchain_core.utils import get_from_env
from langchain_core.embeddings import Embeddings
@classmethod
def from_es_connection(cls, model_id: str, es_connection: Elasticsearch, input_field: str='text_field') -> ElasticsearchEmbeddings:
    """
        Instantiate embeddings from an existing Elasticsearch connection.

        This method provides a way to create an instance of the ElasticsearchEmbeddings
        class using an existing Elasticsearch connection. The connection object is used
        to create an MlClient, which is then used to initialize the
        ElasticsearchEmbeddings instance.

        Args:
        model_id (str): The model_id of the model deployed in the Elasticsearch cluster.
        es_connection (elasticsearch.Elasticsearch): An existing Elasticsearch
        connection object. input_field (str, optional): The name of the key for the
        input text field in the document. Defaults to 'text_field'.

        Returns:
        ElasticsearchEmbeddings: An instance of the ElasticsearchEmbeddings class.

        Example:
            .. code-block:: python

                from elasticsearch import Elasticsearch

                from langchain_community.embeddings import ElasticsearchEmbeddings

                # Define the model ID and input field name (if different from default)
                model_id = "your_model_id"
                # Optional, only if different from 'text_field'
                input_field = "your_input_field"

                # Create Elasticsearch connection
                es_connection = Elasticsearch(
                    hosts=["localhost:9200"], http_auth=("user", "password")
                )

                # Instantiate ElasticsearchEmbeddings using the existing connection
                embeddings = ElasticsearchEmbeddings.from_es_connection(
                    model_id,
                    es_connection,
                    input_field=input_field,
                )

                documents = [
                    "This is an example document.",
                    "Another example document to generate embeddings for.",
                ]
                embeddings_generator.embed_documents(documents)
        """
    from elasticsearch.client import MlClient
    client = MlClient(es_connection)
    return cls(client, model_id, input_field=input_field)