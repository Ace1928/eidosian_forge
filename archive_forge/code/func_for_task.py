from typing import TYPE_CHECKING, Any, Type
from optimum.utils.preprocessing.image_classification import ImageClassificationProcessing
from optimum.utils.preprocessing.question_answering import QuestionAnsweringProcessing
from optimum.utils.preprocessing.text_classification import TextClassificationProcessing
from optimum.utils.preprocessing.token_classification import TokenClassificationProcessing
@classmethod
def for_task(cls, task: str, *dataset_processing_args, **dataset_processing_kwargs: Any) -> 'DatasetProcessing':
    return cls.get_task_processor_class_for_task(task)(*dataset_processing_args, **dataset_processing_kwargs)